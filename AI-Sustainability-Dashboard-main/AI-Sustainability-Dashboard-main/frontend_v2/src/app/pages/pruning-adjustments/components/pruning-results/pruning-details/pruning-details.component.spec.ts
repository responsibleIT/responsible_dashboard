import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PruningDetailsComponent } from './pruning-details.component';

describe('PruningDetailsComponent', () => {
  let component: PruningDetailsComponent;
  let fixture: ComponentFixture<PruningDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PruningDetailsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PruningDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
