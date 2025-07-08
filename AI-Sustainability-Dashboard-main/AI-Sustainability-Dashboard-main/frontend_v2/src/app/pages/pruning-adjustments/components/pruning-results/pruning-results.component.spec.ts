import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PruningResultsComponent } from './pruning-results.component';

describe('PruningResultsComponent', () => {
  let component: PruningResultsComponent;
  let fixture: ComponentFixture<PruningResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PruningResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PruningResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
