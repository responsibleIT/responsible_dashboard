import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PruningSettingsComponent } from './pruning-settings.component';

describe('PruningSettingsComponent', () => {
  let component: PruningSettingsComponent;
  let fixture: ComponentFixture<PruningSettingsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PruningSettingsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PruningSettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
